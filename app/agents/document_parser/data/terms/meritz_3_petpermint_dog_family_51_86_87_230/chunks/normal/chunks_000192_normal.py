from langchain_core.documents import Document

chunk = Document(
    page_content=('제4조(보험금의 청구)\n'
 '\uf000 보험수익자는 다음의 서류를 제출하고 보험금을 청구하 여야 합니다.\n'
 '① 청구서(회사 양식) ② 사고증명서(동물병원 진료비 영수증(진료 항목별 영수 금액 포함), 동물병원 진료기록부, X-ray 등 방사선 '
 '촬영을 하는 경우 해당 사진(촬영일자 및 시간 필수),'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 91},
 'term_type': 'special',
 'clause': {'clause_type': 'claim', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000192',
              'chunk_char_len': 158,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
