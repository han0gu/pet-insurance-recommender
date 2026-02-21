from langchain_core.documents import Document

chunk = Document(
    page_content=('- 도 회사는 이를 지급하지 않습니다.\n'
 '- 제6조 (보험금의 청구)\n'
 '지정대리청구인은 회사가 정하는 방법에 따라 다음의 서류를 제출하고 보험금을 청구하\n'
 '여야 합니다.1. 청구서(회사양식)\n'
 '2. 사고증명서(장해진단서, 입원치료확인서 등)\n'
 '3. 신분증(주민등록증이나 운전면허증 등 사진이 붙은 정부기관 발행 신분증)\n'
 '4. 피보험자 및 지정대리청구인의 가족관계등록부(가족관계증명서)\n'
 '5. 기타 지정대리청구인이 보험금의 수령에 필요하여 제출하는 서류- \n'
 '제7조 (준용규정)-'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000757',
              'chunk_char_len': 260,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
