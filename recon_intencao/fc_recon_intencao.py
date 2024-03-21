from transformers import pipeline
import numpy as np
import re


def reconhecer_intencao(texto):
    """
    Essa função recebe um texto e retorna uma tupla com 7 posições:

    Em caso de SUCESSO na execução e SUCESSO no reconhecimento de intenção:
    1- sucesso na execucao = True
    2- sucesso no reconhecimento: True
    3- acao: string indicando a ação identificada
    4- confianca_acao: float indicando a confiança da ação identificada
    5- assunto: string indicando o assunto identificado
    6- confianca_assunto: float indicando a confiança do assunto identificado
    7- auxiliar: string complementando a informação do assunto (ex: tempo em dias)

    Em caso de SUCESSO na execução e FALHA no reconhecimento de intenção:
    1- sucesso na execucao = Trueprocessos
    2- sucesso no reconhecimento: False
    3- acao: string indicando a intenção genérica do texto
    4- confianca_acao: ''
    5- assunto: ''
    6- confianca_assunto: ''
    7- auxiliar: ''


    Em caso de FALHA na execução, a função retorna:
    1- sucesso na execucao: False
    2- sucesso no reconhecimento: False
    3- acao: string com erro python
    4- confianca_acao: ''
    5- assunto: ''
    6- confianca_assunto: ''
    7- auxiliar: ''
    """

    def recuperar_intecao_nao_identificada(texto):
        QA_input = {'question': 'Descreva a intenção, em detalhes?', 'context': texto}
        res = pipe_qa(QA_input)
        retorno = f'Entendemos que você deseja "{res["answer"]}", mas esta funcionalidade não está disponível.'
        return retorno

    def recuperar_acao_de_texto(texto):
        sucesso = False
        candidate_labels_acao = ['encaminhar', 'enviar', 'mandar', 'tramitar',
                                'receber', 'querer']
        ret = pipe_zsc(texto, candidate_labels_acao)
        #print(ret)
        label = ret['labels'][np.argmax(ret['scores'])]
        confidence = ret['scores'][np.argmax(ret['scores'])]

        if confidence < LIMIAR:
            acao = recuperar_intecao_nao_identificada(texto)
        else:
            if label in ('encaminhar', 'enviar', 'mandar', 'tramitar'):
                acao = 'encaminhar'
            elif label in ('querer', 'receber'):
                acao = 'receber'
            sucesso = True
        return (sucesso, acao, confidence)

    def converter_string_tempo_para_dias(tempo):
        if 'dias' in tempo:
            return re.sub('[^0-9]', '', tempo)
        elif 'meses' in tempo:
            return str(int(re.sub('[^0-9]', '', tempo)) * 30)
        elif 'semanas' in tempo:
            return str(int(re.sub('[^0-9]', '', tempo)) * 7)


    def recuperar_assunto_de_texto(texto):
        tempo_em_dias = ''
        candidate_labels_assunto = ['maria da penha', 'lei maria da penha', 'lei 11.340/2006'
                                    'detalhes', 'informações',
                                    'processos abertos', 'processos conclusos', 'processos aguardando decisão judicial']
        ret = pipe_zsc(texto, candidate_labels_assunto)
        label = ret['labels'][np.argmax(ret['scores'])]
        confidence = ret['scores'][np.argmax(ret['scores'])]
        if confidence < LIMIAR:
            assunto = recuperar_intecao_nao_identificada(texto)
            sucesso = False
        else:
            sucesso = True
            if label in ('maria da penha', 'lei maria da penha', 'lei 11.340/2006'):
                assunto = 'maria da penha'
            elif label in ('processos abertos', 'processos conclusos', 'processos aguardando decisão judicial'):
                assunto = 'processos abertos'
                QA_input = {'question': 'Quanto tempo?', 'context': texto}
                res = pipe_qa(QA_input)
                tempo_em_dias = converter_string_tempo_para_dias(res['answer'])

        return (sucesso, assunto, confidence, tempo_em_dias)

    try:
        LIMIAR = 0.25
        pipe_zsc = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')
        pipe_qa = pipeline("question-answering", model='deepset/roberta-base-squad2')

        ret_acao, acao, confianca_acao = recuperar_acao_de_texto(texto)
        if ret_acao:
            ret_assunto, assunto, confianca_assunto, tempo = recuperar_assunto_de_texto(texto)
            if ret_assunto:
                return (True, True, acao, confianca_acao, assunto, confianca_assunto, tempo)
            else:
                intencao_NI = assunto
                return (True, False, intencao_NI, '', '', '', '')
        else:
            intencao_NI = acao
            return (True, False, intencao_NI, '', '', '', '')

    except Exception as e:
        return (False, False, e, '', '', '', '')


if __name__ == '__main__':
    try:
        #text = 'Encaminhar o processo 0717295-79.2020.8.07.0023 para o Tribunal de Justiça do Estado de São Paulo' # mandar processo  - confidence: 0.276429146528244
        #text = 'Quero detalhes do processo 0717295-79.2020.8.07.0023'                   # detalhes do processo  - confidence: 0.2964492738246918
        #text = 'Me passe informações do processo 0717295-79.2020.8.07.0023'             # detalhes do processo  - confidence: 0.27180159091949463
        #text = 'Quero enviar o processo 0717295-79.2020.8.07.0023 para o cartório'      # enviar processo  - confidence: 0.30082789063453674
        #text = 'Me envie os processos relativos a maria da penha'                       # enviar processo  - confidence: 0.15193483233451843
        #text = 'Como imprimir um texto em python?'                                      # encaminhar processo  - confidence: 0.1102592796087265'
        #text = 'Documentos relativos a Maria da Penha'                                  # maria da penha  - confidence: 0.3441862463951111
        #text = 'Quero receber um relatorio dos processos abertos a 100 dias'
        text = 'Quero um relatório dos processos abertos há 3 meses'
        #text = 'Quais são os processos conclusos há 2 semanas?'
        #text = 'Quais processos estão aguardando decisão judicial há 90 dias'

        #retorno = reconhecer_intencao(text)
        retorno = model_qa(text, 'Quanto tempo?')
        print(retorno)
    except Exception as e:
        print('Erro:', e)
