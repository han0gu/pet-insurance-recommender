from langchain_core.documents import Document

chunk = Document(
    page_content=('② 계약자 또는 피보험자가 제1항 각 호의 통지를 게을리하여 손해가 증가된 때에는 회 사는 그 증가된 손해를 보상하지 않으며, 제1항 '
 '제3호의 통지를 게을리 한 때에는 소 송비용과 변호사비용도 보상하지 않습니다. 다만, 계약자 또는 피보험자가 상법 제 657조 제1항에 '
 '의해 보험사고의 발생을 회사에 알린 경우에는 제3조(보상하는 손해) 제2항 제1호 및 제2호 다.목 또는 라.목의 비용에 대하여 '
 '보상한도액 내에서 보상합니 다.\n'
 '제7조 (보험금의 지급한도)'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 88},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000571',
              'chunk_char_len': 256,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
