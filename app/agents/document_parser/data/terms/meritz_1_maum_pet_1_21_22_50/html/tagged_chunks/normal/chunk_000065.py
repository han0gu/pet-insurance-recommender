from langchain_core.documents import Document

chunk = Document(
    page_content=('. 그러나 계약자, 피보험자 또는 보험수익자의<br>책임 있는 사유로 지급이 지연된 때에는 그 해당기간에 대한 이자는 더하여 '
 '지급하지<br>않습니다.<br>⑤ 계약자, 피보험자 또는 보험수익자는 제2항의 보험금 지급사유조사와 관련하여 동물병<br>원 등 '
 '의료기관, 경찰서 등 관공서에 대한 회사의 서면에 의한 조사요청에 동의하여야<br>합니다'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000065',
              'chunk_char_len': 191,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
