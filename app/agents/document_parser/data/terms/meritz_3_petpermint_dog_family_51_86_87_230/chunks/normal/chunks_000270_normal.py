from langchain_core.documents import Document

chunk = Document(
    page_content=('\uf000 회사는 계약자의 해약환급금 청구권에 대한 강제집행, 담보권실행, 국세 및 지방세 체납처분절차에 따라 계약이 해지된 경우 해지 '
 '당시의 보험수익자가 계약자의 동의를 얻 어 계약 해지로 회사가 채권자에게 지급한 금액을 회사에 지급하고 제13조(계약내용의 변경 등) '
 '제1항의 절차에 따라 계약자 명의를 보험수익자로 변경하여 계약의 특별부활(효 력회복)을 청약할 수 있음을 보험수익자에게 통지하여야 합 '
 '니다. \uf000 회사는 제1항에 따른 계약자 명의변경 신청 및 계약의 특별부활(효력회복) 청약을 승낙합니다'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 106},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000270',
              'chunk_char_len': 278,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
