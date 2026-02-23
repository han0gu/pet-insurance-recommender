from langchain_core.documents import Document

chunk = Document(
    page_content=('- 라. 보험증권상 보상한도액 내의 금액에 대한 공탁보증보험료. 그러나 회사는 그러\n'
 '- 한 보증을 제공할 책임은 부담하지 않습니다.\n'
 '- 마. 피보험자가 제9조(손해배상청구에 대한 회사의 해결) 제2항 및 제3항의 회사의\n'
 '- 요구에 따르기 위하여 지출한 비용\n'
 '<용어풀이># [유익하였던 비용]물건의 개량∙이용을 위하여 지출되는 비용으로, 물건의 가치를 증가시키는 데 도움이 되는 비용을\n'
 '말합니다.# [공탁보증보험료]가압류, 가집행, 가처분 등 각종 민사사건을 신청할 때, 잘못된 신청으로 인해 발생하는 피신청인'),
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
 'indexing': {'chunk_id': 'chunk_000640',
              'chunk_char_len': 284,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
