from langchain_core.documents import Document

chunk = Document(
    page_content=('- 4. 전쟁, 외국의 무력행사, 혁명, 내란, 사변, 폭동\n'
 '- 73 -② 「응급실」 이 아닌 곳에서 진료를 받은 경우 회사는 보험금을 지급하지 않습니다.# 제6조 (보험금의 청구)① 보험수익자는 '
 '다음의 서류를 제출하고 보험금을 청구하여야 합니다.1. 청구서(회사양식)\n'
 '2. 사고증명서(응급실기록지 사본, 의사소견서 또는 진단서 등)\n'
 '3. 신분증(주민등록증이나 운전면허증 등 사진이 붙은 정부기관 발행 신분증, 본인이\n'
 '아닌 경우에는 본인의 인감증명서, 본인서명사실확인서 또는 안전성과 신뢰성이 확보'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000318',
              'chunk_char_len': 278,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
