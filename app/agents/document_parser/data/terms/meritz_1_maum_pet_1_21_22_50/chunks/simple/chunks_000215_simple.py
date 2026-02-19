from langchain_core.documents import Document

chunk = Document(
    page_content=('단체계약 보험료정산 추가특별약관\n'
 '제1조(보험료의 정산)\n'
 '① 회사는 단체계약 특별약관(이하“특별약관”이라 합니다) 제4조(보험의 목적의 증가 감소 또는 교체) 제2항 및 보통약관 제16조(계약 '
 '후 알릴 의무) 제2항에도 불구하고 이 추가 특별약관에 따라 보험료를 정산합니다. ② 회사는 특별약관 제4조(보험의 목적의 증가 감소 '
 '또는 교체) 제3항에도 불구하고 보험 료가 정산되기 이전일지라도 새로이 증가 또는 교체된 보험의 목적에 대해 생긴 손해 를 보상합니다.\n'
 '제2조(보험의 목적의 명부)'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 39},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000215',
              'chunk_char_len': 273,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
