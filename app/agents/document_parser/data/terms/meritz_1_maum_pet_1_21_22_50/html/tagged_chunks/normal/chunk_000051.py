from langchain_core.documents import Document

chunk = Document(
    page_content=('. 청구서(회사양식)<br>2. 사고증명서(동물병원 진료비 영수증, 동물병원 진료비세부내역서(진료 항목별 영수금<br>액 포함), '
 '동물병원 진료기록부, X-ray 등 방사선 촬영을 하는 경우 해당 사진(촬영<br>일자 및 시간 필수) 등)<br>3'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000051',
              'chunk_char_len': 138,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
