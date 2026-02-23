from langchain_core.documents import Document

chunk = Document(
    page_content=('. 서혜부허니아(서혜부탈장), 첩모난생(속눈썹 질환), 눈물샘으로 인한 비용<br>13. 입원중의 식이(食餌)에 해당하지 않는 음식물 및 '
 '식이요법, 수의사 처방 의약품 이<br>외의 것(건강보조 식품, 의약품지정이 되어 있지 않은 한방약, 의약부외품 등)<br>14. '
 '한의학(단, 침술을 제외합니다.), 인도 의학, 허브요법, 아로마테라피 등의 대체의료,<br>재활치료<br>15'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive', 'eye']},
 'indexing': {'chunk_id': 'chunk_000044',
              'chunk_char_len': 212,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
