from langchain_core.documents import Document

chunk = Document(
    page_content=('657조 제1항에 의해 보험사고의 발생을 회사에 알린 경우에는 제3조(보상하는 손해)\n'
 '제2항 제1호 및 제2호 다.목 또는 라.목의 비용에 대하여 보상한도액 내에서 보상합니\n'
 '다.# 제7조 (보험금의 지급한도)① 회사는 제3조(보상하는 손해) 제2항의 손해에 대하여 다음과 같이 보상합니다. 이 경\n'
 '우 보상한도액과 자기부담금은 각각 보험증권에 기재된 금액을 말합니다.1. 제3조(보상하는 손해) 제2항 제1호의 손해배상금 : 매회의 '
 '사고마다 자기부담금 10\n'
 '만원을 초과하는 경우에 한하여 그 초과하는 배상책임 손해에 대한 금액을 보상한'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000493',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
