# Projetos com Feedback
# Formação Cientista de Dados
# Curso Business Analytics
# Sistema de Recomendação Para Rede de Varejo Usando Market Basket Analysis
# Objetivo do Projeto: prever quais produtos adquiridos anteriormente estarão no próximo pedido de um usuário

# ------------------------------
# Carregando Pacotes Necessários
# ------------------------------
library(data.table)
library(dplyr)
library(arules)
library(ggplot2)
library(gridExtra)
library(wordcloud)
library(xgboost)
library(naniar)
library(caret)
library(RColorBrewer)

# -------------------------------------------
# Leitura dos datasets e Dicionários de Dados
# -------------------------------------------

# Tabela de Pedidos - 3.4 milhões de linhas - Tabela Fato
pedidos <- fread("datasets/orders.csv")

orders_dic <- data.frame(Variável = c("ID Pedido", "ID Comprador", 'Classificação do pedido',
                                      "Número do Pedido", "Dia do Pedido", "Hora do Pedido",
                                      "Dias Desde o Último Pedido"),
                         Descrição = c("identificação do pedido", "identificação do comprador",
                         "classificação de pedido: prior = pedido comum/ train = pedidos reservados para dataset de treino/ test = pedidos reservados para dataset de teste",
                         "nº do pedido realizado pelo consumidor",
                         "dia da semana que o pedido foi realizado pelo consumidor",
                         "hora do dia que o pedido foi realizado",
                         "dias desde o último pedido deste consumidor"))

View(orders_dic)
View(pedidos)

# Tabela de Produtos - 50 mil linhas - Tabela Dimensão
produtos <- fread("datasets/products.csv")

produtos_dic <- data.frame(Variável = c("ID Produto", "Nome do Produto",
                                        "ID do Corredor", "ID do Departamento"),
                           Descrição = c("identificação do produto", "nome do produto",
                                         "identificação do corredor que o produto está localizado.",
                                         "identificação do departamento que o produto pertence."))
View(produtos_dic)
View(produtos)

# Tabela de Corredores - 134 linhas - Tabela Dimensão
corredores <- fread("datasets/aisles.csv")

corredores_dic <- data.frame(Variável = c("ID Corredor", "Nome do Corredor"),
                             Descrição = c("identificação dos corredores do mercado.",
                                           "nomes dos corredores do mercado."))
View(corredores_dic)
View(corredores)

# Tabela de Departamentos - 21 linhas - Tabela Dimensão
departamentos <- fread("datasets/departments.csv")

departamentos_dic <- data.frame(Variável = c("ID Departamento", "Nome do Departamento"),
                                Descrição = c("identificação dos departamentos do mercado.",
                                              "nome dos departamentos do mercado."))

View(departamentos_dic)
View(departamentos)

# Tabelas de Especificidades de Pedidos - >= 30 milhões de linhas
# Carregando apenas 'order_products__train.csv' para aplicar a regra de associação com ele por conta de memória
pedidos_esp <- fread("datasets/order_products__train.csv")

pedidos_esp_dic <- data.frame(Variável = c("ID Pedido", "ID Produto",
                                           "Ordem de Adição ao Carrinho", "Produto Repetido?"),
                              Descrição = c("identificação do pedido", "identificação do produto",
                                            "ordem que o item foi adicionado ao carrinho em determinada compra",
                                            "informa se o produto está sendo pedido novamente (1) ou não (0)."))
View(pedidos_esp)
View(pedidos_esp_dic)

# -----------------------
# Tratamento dos datasets
# -----------------------

# Fazendo cópia dos datasets para partes 1 e 2
# (boa prática caso seja necessário voltar ao dataset original por algum motivo)
df_pedidos <- pedidos
df_pedidos2 <- pedidos

# Valores NA
na_count <- as.data.frame(sapply(df_pedidos, function(y) sum(length(which(is.na(y))))))
names(na_count) <- "Qtd. NAs"

View(na_count)

# Observamos que existe um número considerável de NAs, porém, apenas na coluna referente a quantidade de dias desde a última compra de determinado consumidor.
# Para tratar esses valores NAs, vamos substituir esses NAs pelo número 0, tendo em vista que se não houve intervalo de dias entre as compras, podemos aplicar o número zero para representar esta informação.

df_pedidos$days_since_prior_order[is.na(df_pedidos$days_since_prior_order)] <- 0
df_pedidos2$days_since_prior_order[is.na(df_pedidos2$days_since_prior_order)] <- 0

# -------
# Parte 1
# -------
# Análise Exploratória com identificação de itens mais pedidos em conjunto (duplas)
# Foco na exploração do dataset com pacote dlpyr e depois aplicação dos algoritmos de MBA
# (Market Basket Analysis) ECLAT e APRIORI.

# -------------------
# Feature Engineering
# -------------------

# Dias da Semana
# Criando nova coluna (dia_da_semana) com nome do dia da semana para posterior análise exploratória

df_pedidos$dia_da_semana[df_pedidos$order_dow == 0] <- "Domingo"
df_pedidos$dia_da_semana[df_pedidos$order_dow == 1] <- "Segunda"
df_pedidos$dia_da_semana[df_pedidos$order_dow == 2] <- "Terça"
df_pedidos$dia_da_semana[df_pedidos$order_dow == 3] <- "Quarta"
df_pedidos$dia_da_semana[df_pedidos$order_dow == 4] <- "Quinta"
df_pedidos$dia_da_semana[df_pedidos$order_dow == 5] <- "Sexta"
df_pedidos$dia_da_semana[df_pedidos$order_dow == 6] <- "Sábado"

# Transformando coluna de dia da semana em fator
df_pedidos$dia_da_semana <- factor(df_pedidos$dia_da_semana,
                                   levels = c("Domingo", "Segunda", "Terça", "Quarta", "Quinta", "Sexta", "Sábado"))

# Periodicidade de Compras
sort(unique(df_pedidos$days_since_prior_order))

# Se days_since_prior = 0, então nenhuma, ou seja, o usuário não fez nenhum outro pedido na plataforma
# Se days_since_prior entre 1 e 15 inclusive, então média, ou seja, o usuário fez de 1 até 15 pedidos na plataforma
# Se days_since_prior > 15 exclusive, então alta, ou seja, o usuário realizou mais de 15 pedidos na plataforma

df_pedidos$periodicidade[df_pedidos$days_since_prior_order == 0] <- "baixa"
df_pedidos$periodicidade[df_pedidos$days_since_prior_order > 15] <- "alta"
df_pedidos$periodicidade[is.na(df_pedidos$periodicidade)] <- "média"

# Conferindo se alguma das transformações gerou valores NA antes de seguir para análise exploratória
sum(is.na(df_pedidos))

View(df_pedidos)

# Salvando df
write.csv(df_pedidos, file = "df_novo/df_pedidos.csv", row.names = F, quote = F)

# Tratamento de dataset pedidos_esp para os algoritmos de MBA
# O dataset pedidos_esp será combinado com demais tabelas dimensão
# e será utilizado, além de em alguns momentos para análise exploratória,
# prioritariamente para utilização no ECLAT e no APRIORI

pedidos_esp_analise <- pedidos_esp %>%
    left_join(produtos, by = "product_id")

View(pedidos_esp_analise)

df_modelo <- pedidos_esp_analise %>%
    group_by(order_id) %>%
    summarise(produtos = as.vector(list(product_name)))

View(df_modelo)

# --------------------
# Análise Exploratória
# --------------------

# Para construção dos gráficos usarei a escala Greys do pacote RColorBrewer,
# com o acréscimo de uma cor para ter uma paleta com 10 cores

cores <- c(brewer.pal(n = 9, "Greys"),"#999999")

# --------------------- Dataset df_pedidos ---------------------

# Quantidade de pedidos por hora do dia

quadro1 <- df_pedidos %>%
    group_by(order_hour_of_day) %>%
    summarise(contagem = n())

media1 <- round(mean(c(quadro1$contagem[quadro1$order_hour_of_day <= 15],
                quadro1$contagem[quadro1$order_hour_of_day <= 15])),0)

#png("graficos/qtd_ped_hora.png", width = 10, height = 6, units = "in", res = 200)
# tamanho da letra do texto com a média do pico - size = 5

df_pedidos %>%
    group_by(order_hour_of_day) %>%
    summarise(contagem = n()) %>%
    ggplot(aes(x = order_hour_of_day, y = contagem, group = 1)) + geom_line(size = 1) +
    geom_text(aes(label = paste("média no pico =", media1), x = 12.7, y = 255000), colour = "#B15928", size = 10) +
    labs(title ='Quantidade de Pedidos por Hora', x = '', y = '') +
    scale_x_continuous(labels = paste0(seq(0, 25, 5), 'h')) +
    scale_y_continuous(limits = c(5000, 300000), breaks = seq(5000, 300000, 50000)) +
    theme(plot.title = element_text(hjust = 0.5))

#dev.off()

# Análise: Percebemos um pico de pedidos entre 10h e 15h, que se mantém estável e
# inicia queda por volta das 16h. Vale atentar para reposição de produtos,
# atendimento e instabilidades do sistema neste período de tempo.

# Gráfico de colunas com qtd de clientes com periodicidade baixa, média e alta

#png("graficos/periodicidde.png", width = 10, height = 6, units = "in", res = 200)

df_pedidos %>%
    group_by(periodicidade) %>%
    summarise(contagem = n()) %>%
    mutate(percentual = paste(round(contagem / sum(contagem) * 100,2), "%")) %>%
    ggplot(aes(x = reorder(periodicidade, - contagem), y = contagem)) + geom_col(aes(fill = periodicidade)) +
    geom_text(aes(label = percentual), vjust = -1, colour = "#B15928") +
    labs(title ='Periodicidade de Clientes', x = '', y = '') +
    theme(plot.title = element_text(hjust = 0.5), legend.position="none") +
    scale_fill_manual(values = cores[8:10])
    

#dev.off()

# Análise: Em sua maioria, de acordo com o conceito periodicidade que definimos anteriormente, 
# o perfil dos usuários deste conjunto de dados é de consumo médio, ou seja, realizaram
# de 1 a 15 pedidos no período de tempo analisado.
# Outro aspecto a ressaltar é que os clientes com periodicidades "extremas" (alta e baixa) correspondem
# a menos de 32% do total de clientes deste dataset.

# Gráfico de barras com corredores e produtos mais acessados

#png("graficos/prod_cor.png", width = 10, height = 6, units = "in", res = 200)

quadro_graf1 <- pedidos_esp %>%
    left_join(produtos, by = "product_id") %>%
    group_by(product_name) %>%
    summarise(contagem = n()) %>%
    mutate(product_name = reorder(product_name, contagem)) %>%
    slice_max(product_name, n = 10)

media_graf1 <- mean(quadro_graf1$contagem)

graf1 <- pedidos_esp %>%
    left_join(produtos, by = "product_id") %>%
    group_by(product_name) %>%
    summarise(contagem = n()) %>%
    mutate(product_name = reorder(product_name, contagem)) %>%
    slice_max(product_name, n = 10) %>%
    ggplot(aes(x = product_name, y = contagem)) + geom_col(aes(fill = product_name)) +
    geom_hline(yintercept = media_graf1, linetype = "dashed", size = 1, colour = "#B15928") + 
    coord_flip() + labs(x = 'Produtos', y = '') +
    theme(legend.position="none") +
    scale_fill_manual(values = cores)

quadro_graf2 <- pedidos_esp %>%
    left_join(produtos, by = "product_id") %>%
    left_join(corredores, by = "aisle_id") %>%
    group_by(aisle) %>%
    summarise(contagem = n()) %>%
    mutate(aisle = reorder(aisle, contagem)) %>%
    slice_max(aisle, n = 10)

media_graf2 <- mean(quadro_graf1$contagem)

graf2 <- pedidos_esp %>%
    left_join(produtos, by = "product_id") %>%
    left_join(corredores, by = "aisle_id") %>%
    group_by(aisle) %>%
    summarise(contagem = n()) %>%
    mutate(aisle = reorder(aisle, contagem)) %>%
    slice_max(aisle, n = 10) %>%
    ggplot(aes(x = aisle, y = contagem)) + geom_col(aes(fill = aisle)) + coord_flip() +
    geom_hline(yintercept = media_graf2, linetype = "dashed", colour = "#B15928", size = 1) + 
    labs(x = 'Corredores', y = '') +
    theme(legend.position="none") +
    scale_fill_manual(values = cores)

grid.arrange(graf1, graf2, ncol = 2)

#dev.off()

# Análise: Os produtos mais pedidos neste conjunto de consumidores são frutas.
# Logicamente, os corredores mais visitados são, prioritariamente, os de frutas e vegetais frescos
# e orgânicos, seguidos de outros corredores nos quais você pode encontrar este tipo de 
# produto ou similares (iogurtes e queijos)
# Outro ponto importante é que, observando o gráfico de produtos, apenas os 4 primeiros ultrapassaram
# a média de quantidade de compras, já em relação ao gráfico de corredores, todos os 10 primeiros pelo
# menos foram acessados mais do que a média da selação.

# Qtd de vendas por dia (identificação de dias com picos - acrescentar linha)

quadro2 <- df_pedidos %>%
    group_by(dia_da_semana) %>%
    summarise(contagem = n() / 10)

min(quadro2$contagem)
max(quadro2$contagem)

#png("graficos/venda_dia_semana.png", width = 10, height = 6, units = "in", res = 200)

df_pedidos %>%
    group_by(dia_da_semana) %>%
    summarise(contagem = n()/10) %>%
    ggplot(aes(x = dia_da_semana, y = contagem)) + geom_col(aes(fill = dia_da_semana)) +
    geom_line(aes(y = contagem, group = 1), size = 1, colour = "#B15928") +
    labs(title ='Quantidade de Vendas por Dia', x = '', y = 'Vendas (qtd.)') +
    scale_y_continuous(name = "Vendas (qtd.)", label = paste0(seq(0, 60100, 20000), '0'),
                       limits = c(0, 60100), breaks = seq(0, 60100, 20000)) +
    theme(plot.title = element_text(hjust = 0.5), legend.position="none") +
    scale_fill_manual(values = cores[10:4])

#dev.off()

# Análise: De acordo com os dados deste dataset, há mais pedidos no domingo e na segunda, enquanto nos
# próximos dias parece haver uma média constante, retomando levamente o crescimento na sexta.

# --------------------- Dataset pedidos_esp ---------------------

# Mapa de palavras dos 200 produtos com maior reincidência em pedidos

#png("graficos/mapa_palavras.png", width = 10, height = 6, units = "in", res = 200)

df_wc <- pedidos_esp %>%
    left_join(produtos, by = "product_id") %>%
    group_by(product_name) %>%
    summarise(soma = sum(reordered)) %>%
    mutate(product_name = reorder(product_name, soma)) %>%
    slice_max(product_name, n = 200)



wordcloud(words = df_wc$product_name, freq = df_wc$soma, scale = c(6,1),
          random.order = FALSE, random.color = FALSE , rot.per = 0.35,
          ordered.colors = TRUE, colors = rep(cores[10:6], 40))
#dev.off()

# Análise: Observamos o mesmo padrão, então, neste mapa de palavras, encontramos com bastante
# frequência produtos do tipo frutas, legumes e verduras.
# O que vemos de diferente são alguns produtos que diferem levemente, mas naõ totalmente
# do grupo observado anteriormente, como leite, macarrão e manteiga.
# Aparentemente as pessoas realizam compras para cozinhar refeições, nas quais necessitam
# produtos complementares, além dos originais.

# Tabela de produtos que mais foram adicionados primeiro ao carrinho
prod_table <- pedidos_esp %>%
    filter(add_to_cart_order == 1) %>%
    left_join(produtos, by = "product_id") %>%
    group_by(product_name) %>%
    summarise(contagem = n()) %>%
    mutate(product_name = reorder(product_name, contagem))

prod_table_20 <- pedidos_esp %>%
    filter(add_to_cart_order == 1) %>%
    left_join(produtos, by = "product_id") %>%
    group_by(product_name) %>%
    summarise(contagem = n()) %>%
    mutate(product_name = reorder(product_name, contagem)) %>%
    slice_max(product_name, n = 20)

View(prod_table_20)

# Salvando Tabela Completa
write.csv(prod_table, file = "df_novo/prod_table.csv", row.names = F, quote = F)

# Análise: Nesta tabela observamos, também, a presença de alimentos específicos e também perecíveis,
#  ou seja, de pronto consumo. Isso pode significar que o consumidor, alem de  se preocupar com o preparo
#  de suas refeições e, ao realizar as compras, pretende elaborar seus pratos de maneira imediata.

# -----------------------------------
# Aplicação dos Algoritmos de MBA (ECLAT e APRIORI)
# -----------------------------------

# Join com dataset de produtos para coletar nome de cada produto

df_modelo <- pedidos_esp %>%
    left_join(produtos, by = "product_id")

# Algoritmos de Regra de Associação e seus detalhes
# O que é um algoritmo de regra de associação?
# Suporte = conta quantas vezes o item aparece na relação de proutos
# Confiança = é o cálculo da probabilidade da ocorrência de uma dupla de produtos 
# Elevação = é a medida que informa quanto um item X aumenta a possibilidade de ter o item Y
# no mesmo pedido. Para elevações > 1, sabemos que ter X no pedido leva também a pedir Y, já
# elevações < 1 nos informa que pedir X não significa aumento de chances de pedir também Y,
# mesmo que a confiança entre os 2 itens esteja alta.

# Transformação
# Estes algoritmos esperam receber uma lista de listas para cada transação (pedido), então, realizarei um agrupamento de acordo com o id do pedido, apresentando todos os itens comprados em cada pedido

df_modelo <- df_modelo %>%
    group_by(order_id) %>%
    summarize(produtos = as.vector(list(product_name)))

# Salvando df pronto para o Modelo (flatten lista na coluna de produtos para conseguir salvar)
df_modelo_save <- apply(df_modelo,2,as.character)
write.csv(df_modelo_save, file = "df_novo/df_modelo.csv", row.names = F, quote = F)

# Transformando o dataset em um objeto do tipo transactions, formando uma matriz esparsa
# e entendendo o resultado

transacoes <- as(df_modelo$produtos, "transactions")

summary(transacoes)

# Plot da distribuição de frequência dos 10 itens que mais aparecem
#png("graficos/transacoes.png", width = 10, height = 6, units = "in", res = 200)
itemFrequencyPlot(transacoes, topN = 10)
#dev.off()

# ECLAT
# O algoritmo ECLAT lida apenas com o suporte, ou seja, leva em consideração apenas a frequência dos itens de acordo com as transações (pedidos, neste caso), sendo uma versão simplificada do algoritmo APRIORI

combinacoes_eclat <- eclat(data = transacoes, parameter = list(support = .003, minlen = 2))

summary(combinacoes_eclat)

# Apresentando os 10 primeiros resultados ordenados pelo suporte
df_eclat <- inspect(sort(combinacoes_eclat, by = "support")[1:10])

View(df_eclat)

# Salvando resultado do algoritmo ECLAT
df_eclat_completo <- inspect(sort(combinacoes_eclat, by = "support")[1:38])
write.csv(df_eclat_completo, file = "df_novo/df_eclat.csv", row.names = F, quote = F)

# APRIORI
# A diferença do APRIORI para o ECLAT é a inclusão do parâmetro de confiança e da elevação no cálculo,
# tornando este um algoritmo mais completo

combinacoes_apriori <- apriori(data = transacoes,
                               parameter = list(supp = .003, conf = .25, minlen = 2))

summary(combinacoes_apriori)

# Apresentando os 10 primeiros resultados ordenados pelo suporte
inspect(sort(combinacoes_apriori, by = "lift")[1:10])

df_apriori <- data.frame(
    lhs = labels(lhs(combinacoes_apriori)),
    rhs = labels(rhs(combinacoes_apriori)),
    combinacoes_apriori@quality)

View(df_apriori)

# Salvando resultado do algoritmo APRIORI
write.csv(df_apriori, file = "df_novo/df_apriori.csv", row.names = F, quote = F)

# ------------------
# Conclusões Parte 1
# ------------------

# Itens mais comprados juntos:
# Neste sentido, parece interessante colocar estes produtos próximos nas gôndolas
# quando lojas físicas ou locais nas páginas quando e-commerce.
# Cabe também ressaltar a importância de se atentar para os horários de picos de venda,
# também analisados neste trabalho, para que não falte produtos
# nos horários que os consumidores mais precisam.

# -------
# Parte 2
# -------
# Análise Exploratória com fins de previsão de valores
# Fazendo previsões de acordo com o user_id ou order_id - quais os produtos que serão comprados juntos?

# Join dos datasets
# Retiradas as colunas de dia da semana e hora da realização do pedido e número do pedido
# pois não agregam valor a esta parte do projeto

novo_df <- df_pedidos2 %>%
    select(-order_dow, -order_hour_of_day, -order_number) %>%
    left_join(pedidos_esp, by = c("order_id"))

# -------------------
# Feature Engineering
# -------------------

# Quantidade de produtos naquela compra

novo_df <- novo_df %>%
    left_join(novo_df %>% group_by(order_id) %>% summarize(qtd_prod_por_pedido = n()),
              by = c ("order_id"))

# Quantidade de vezes que o produto foi pedido

novo_df <- novo_df %>%
    left_join(novo_df %>% group_by(product_id) %>% summarize(qtd_vezes_prod_pedido = n()),
              by = c("product_id"))

# Quantidade de vezes que o produto foi pedido como 1 item do carrinho

novo_df <- novo_df %>%
    left_join(novo_df %>% filter(add_to_cart_order == 1) %>% group_by(product_id) %>% summarize(prim_carrinho = n()),
              by = c("product_id"))

# Quantidade de vezes que o produto foi reorded

novo_df <- novo_df %>%
    left_join(novo_df %>% filter(reordered == 1) %>% group_by(product_id) %>% summarize(qtd_reorded = n()),
              by = c("product_id"))

# Quantidade de pedidos feitos pelo usuário / Média total de pedidos

novo_df <- novo_df %>%
    left_join(novo_df %>% group_by(user_id) %>% summarize(contagem = n())
              %>% mutate(media_ped_usu = round(contagem / mean(contagem),2)) %>% select(-contagem),
              by = c("user_id"))

# Substituindo valores NA
#png("graficos/valores_nulos.png", width = 10, height = 6, units = "in", res = 200)

gg_miss_var(novo_df, show_pct = TRUE) +
    labs(x = "Variáveis", y = "%", title = "Valores Nulos") +
    theme(plot.title = element_text(hjust = 0.5)) + ylim(0, 100)

#dev.off()
# As colunas prim_carrinho, qtd_reorded e add_to_cart_order serão preenchidas com zeros
# e as colunas reorded e product_id se manterão com NAs pois serão deletadas do dataset antes
# do início do treinamento do modelo

novo_df$prim_carrinho[is.na(novo_df$prim_carrinho)] <- 0
novo_df$qtd_reorded[is.na(novo_df$qtd_reorded)] <- 0
novo_df$add_to_cart_order[is.na(novo_df$add_to_cart_order)] <- 0

# Conferindo se temos apenas reorded e product_id com valores nulos

df_nulo <- as.data.frame(sapply(novo_df, function(y) sum(length(which(is.na(y))))))
names(df_nulo) <- "Soma de Nulos"

# Salvando df final para as previsões
write.csv(novo_df, file = "df_novo/df_pedidos2.csv", row.names = F, quote = F)

# ---------------------------------
# Trainamento e Avaliação do Modelo
# ---------------------------------

# Divisão do dataset onde eval_set == 'train' entre treino e teste

treino <- novo_df %>%
    filter(eval_set == "train") %>%
    select(-order_id, -eval_set)

divisao <- createDataPartition(treino$reordered, p = .8, list = FALSE)
df_treino <- treino[divisao,]
df_teste <- treino[-divisao,]

# Criação de matriz xgb
mat_treino <- xgb.DMatrix(as.matrix(df_treino %>% select(-reordered, -product_id, -user_id))
                   ,label = df_treino$reordered)

modelo <- xgboost(dat = mat_treino, nrounds = 50)

# Salvando modelo
xgb.save(modelo, fname = "modelo")

# Aplicação do modelo no grupo de teste
mat_teste <- xgb.DMatrix(as.matrix(df_teste %>% select(-reordered, -product_id, -user_id))
                         ,label = df_teste$reordered)

df_teste$previsoes <- predict(modelo, mat_teste)

# Como o algoritimo xgboost nos entrega uma probabilidade, ou seja, se a previsão foi de 62%,
# isto significa que existe 62% de chances daquele produto ser pedido novamente, vamos
# aplicar um limiar (threshold) para selecionar apenas previsões >= 70%

df_teste$previsoes <- ifelse(df_teste$previsoes >= .7, 1, 0)

# Avaliação do modelo
# Para saber se o modelo foi bem sucedido, vou verificar se as previsoes acertaram se comparadas
# às verdadeiras classificações do dataset

df_teste$sucesso <- ifelse(df_teste$reordered == df_teste$previsoes, "sim", "não")

df_teste %>%
    group_by(sucesso) %>%
    summarize(contagem = n()) %>%
    mutate(`% de acertos` = paste0(round(((contagem / sum(contagem)) * 100), 2), " %"))

# Nosso modelo acertou pouco mais 65% dos casos, não sendo muito satisfatório.
# Vamos tentar alcançar pelo menos 70% com um novo treinamento com mais rounds

novo_modelo <- xgboost(dat = mat_treino, nrounds = 70)

# Salvando novo modelo
xgb.save(novo_modelo, fname = "novo_modelo")

df_teste$previsoes2 <- predict(novo_modelo, mat_teste)

df_teste$previsoes2 <- ifelse(df_teste$previsoes2 >= .7, 1, 0)

# Agora, para avaliar o modelo, criaremos uma matriz de confusão
# A matriz de confusão é uma tabela de contingência de tamanho 2 x 2
# que nos permite identificar quantas vezes o modelo acertou e quantas ele errou

confusionMatrix(as.factor(df_teste$previsoes2), as.factor(df_teste$reordered))

# Não obtivemos resultado muito diferente, a acurácia segue mais ou menos por volta de 65%.
# Na matriz de confusão observamos poucos falsos positivos - quantidade de vezes que o produto
# não foi pedido novamente mas o modelo previu que sim - mas altíssimos casos de falsos negativos -
# quantidade de vezes que o produto foi pedido novamente mas o modelo previu que não.

# ------------------
# Conclusões Parte 2
# ------------------

# Não obtive bons resultados com a aplicação do algoritmo xgboost.
# Cabe realização de mais estudos para aplicações de outros algoritmos ou, a realização
# de nova etapa de feature engineering ou, ainda, o ajuste de hiperparâmetros no algoritmo usado.

# ---------------------------------------------------------------------------------------
# Entrega do Resultado - quais produtos previstos para os usuários em seu próximo pedido?
# ---------------------------------------------------------------------------------------

# Antes de entregar o resultado, vamos comparar os produtos que de fato foram comprados
# e os que nosso modelo previu

previsoes <- df_teste %>%
    filter(previsoes2 == 1) %>%
    inner_join(produtos, by ="product_id") %>%
    group_by(user_id) %>%
    summarize(produtos_previstos = paste0(product_name, collapse = ", "))

produtos_reais <- df_teste %>%
    filter(reordered == 1) %>%
    inner_join(produtos, by ="product_id") %>%
    group_by(user_id) %>%
    summarize(produtos_reais = paste0(product_name, collapse = ", "))

comparando_produtos <- previsoes %>%
    inner_join(produtos_reais, by = "user_id")

View(comparando_produtos)

# Salvando df comparativo de produtos previstos x produtos reais
write.csv(comparando_produtos, file = "df_novo/comparando_produtos.csv", row.names = F, quote = F)

sub <- df_teste %>%
    filter(previsoes2 == 1) %>%
    group_by(user_id) %>%
    summarize(produtos = paste0(product_id, collapse = " "))

write.csv(sub, file = "df_novo/resultado_final.csv", row.names = F, quote = F)

# --------------------------
# Referências Bibliográficas
# --------------------------

# “The Instacart Online Grocery Shopping Dataset 2017”,
# Accessed from https://www.instacart.com/datasets/grocery-shopping-2017 on 2022-01-09

# Formação Cientista de Dados - Data Science Academy