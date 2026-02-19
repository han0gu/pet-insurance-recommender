from langchain_core.documents import Document

chunk = Document(
    page_content=('보험료정산 추가특별약관\n'
 '(단체계약 특별약관에 적용)\n'
 '제1조(보험료의 정산)\n'
 '회사는 단체계약 특별약관 제4조(보험의 목적의 증가 감소 또는 교체) 제2항에도 불구하고 이 추가 특별약관에 따라 보험료를 정산합니다. '
 '② 회사는 단체계약 특별약관 제4조(보험의 목적의 증가 감소 또는 교체) 제3항과 관계없이 보험료가 정 산되기 이전 일지라도 새로이 증가 '
 '또는 교체된 피보험자에 대해 생긴 손해를 보상하여 드립니다.\n'
 '제2조(피보험자의 명부)\n'
 '계약자는 항상 피보험자 명부를 비치하여 회사가 열람을 요구할 경우에는 이에 따라야 합니다.'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 37},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000184',
              'chunk_char_len': 293,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
