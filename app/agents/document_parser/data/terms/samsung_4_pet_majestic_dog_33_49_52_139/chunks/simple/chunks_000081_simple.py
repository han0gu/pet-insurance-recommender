from langchain_core.documents import Document

chunk = Document(
    page_content=('. ⑥ 제1항에도 불구하고 알릴 의무를 위반한 사실이 보험금 지급사유 발생에 영향을 미쳤 음을 회사가 증명하지 못한 경우에는 제4항 및 '
 '제5항에 관계없이 약정한 보험금을 지 급합니다.'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 40},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000081',
              'chunk_char_len': 102,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
