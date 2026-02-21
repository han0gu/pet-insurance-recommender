from langchain_core.documents import Document

chunk = Document(
    page_content=('- 가 정하는 기준에 따라 일부 보험계약의 경우 분납이 제한될 수 있습니다.\n'
 '- ④ 제1항의 통지에 따라 위험의 증가로 보험료를 더 내야 할 경우 회사가 청구한 추가보\n'
 '- 험료(정산금액을 포함합니다)를 계약자가 납입하지 않았을 때, 회사는 위험이 증가되\n'
 '- 기 전에 적용된 보험요율(이하「변경전 요율」이라 합니다)의 위험이 증가된 후에 적\n'
 '- 용해야 할 보험요율(이하「변경후 요율」이라 합니다)에 대한 비율에 따라 보험금을\n'
 '- 삭감하여 지급합니다. 다만, 증가된 위험과 관계없이 발생한 보험금 지급사유에 관해'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000059',
              'chunk_char_len': 284,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
