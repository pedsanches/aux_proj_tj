from transformers import pipeline
import numpy as np
import re

class Recon_Itencao:
    pipe_zsc = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')
    pipe_qa = pipeline("question-answering", model='deepset/roberta-base-squad2')
    LIMIAR = 0.2

    def classifier(self, texto):
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
        try:
            ret_acao, acao, confianca_acao = self.recuperar_acao_de_texto(texto)
            if ret_acao:
                ret_assunto, assunto, confianca_assunto, tempo = self.recuperar_assunto_de_texto(texto)
                if ret_assunto:
                    return {'sucesso_exec': True, 
                        'sucesso_recon': True, 
                        'acao': acao, 
                        'conf_acao': confianca_acao, 
                        'token': assunto, 
                        'conf_token': confianca_assunto, 
                        'aux': tempo}
            
                else:
                    intencao_NI = assunto
                    return {'sucesso_exec': True, 
                        'sucesso_recon': False, 
                        'acao': intencao_NI, 
                        'conf_acao': '', 
                        'token': '', 
                        'conf_token': '', 
                        'aux': ''}
            else:
                intencao_NI = acao
                return {'sucesso_exec': True, 
                        'sucesso_recon': False, 
                        'acao': intencao_NI, 
                        'conf_acao': '', 
                        'token': '', 
                        'conf_token': '', 
                        'aux': ''}
            
        except Exception as e:
            print(e)
            return (False, False, e, '', '', '', '')

    def recuperar_intecao_nao_identificada(self, texto):
        QA_input = {'question': 'Descreva a intenção, em detalhes?', 'context': texto}
        res = self.pipe_qa(QA_input)
        retorno = f'Entendemos que você deseja "{res["answer"]}", mas esta funcionalidade não está disponível.'
        return retorno

    def recuperar_acao_de_texto(self, texto):
        sucesso = False
        candidate_labels_acao = ['encaminhar', 'enviar', 'mandar', 'tramitar',
                                'receber', 'querer', 'criar', 'consultar']
        ret = self.pipe_zsc(texto, candidate_labels_acao)
        label = ret['labels'][np.argmax(ret['scores'])]
        confidence = ret['scores'][np.argmax(ret['scores'])]

        if confidence < self.LIMIAR:
            acao = self.recuperar_intecao_nao_identificada(texto)
        else:
            if label in ('encaminhar', 'enviar', 'mandar', 'tramitar'):
                acao = 'encaminhar'
            elif label in ('querer', 'receber'):
                acao = 'receber'
            elif label in ('criar'):
                acao = 'criar'
            elif label in ('consultar'):
                acao = 'consultar'
            sucesso = True
        return (sucesso, acao, confidence)

    def converter_string_tempo_para_dias(self, tempo):
        if 'dias' in tempo:
            return re.sub('[^0-9]', '', tempo)
        elif 'meses' in tempo:
            return str(int(re.sub('[^0-9]', '', tempo)) * 30)
        elif 'semanas' in tempo:
            return str(int(re.sub('[^0-9]', '', tempo)) * 7)


    def recuperar_assunto_de_texto(self, texto):
        tempo_em_dias = ''
        candidate_labels_assunto = ['maria da penha', 'lei maria da penha', 'lei 11.340/2006',
                                    'detalhes', 'informações',
                                    'processos conclusos', 'autos conclusos',
                                    'processos abertos', 'processos aguardando decisão judicial', 'processo',
                                    'alerta',]
        ret = self.pipe_zsc(texto, candidate_labels_assunto)
        label = ret['labels'][np.argmax(ret['scores'])]
        confidence = ret['scores'][np.argmax(ret['scores'])]
        assunto = ''
        if confidence < self.LIMIAR:
            assunto = self.recuperar_intecao_nao_identificada(texto)
            sucesso = False
        else:
            sucesso = True
            if label in ('maria da penha', 'lei maria da penha', 'lei 11.340/2006'):
                assunto = 'MARIA_DA_PENHA'

            elif label in ('processos abertos', 'processos aguardando decisão judicial', 'processo'):
                assunto = 'QUERY_TYPE'
                QA_input = {'question': 'Quanto tempo?', 'context': texto}
                res = self.pipe_qa(QA_input)
                tempo_em_dias = self.converter_string_tempo_para_dias(res['answer'])

            elif label in ('alerta',):
                assunto = 'ALERT_TYPE'
                
            elif label in ('processos conclusos', 'autos conclusos',):
                assunto = 'AUTOS_CONCLUSOS'
                
        
        return (sucesso, assunto, confidence, tempo_em_dias)

    
