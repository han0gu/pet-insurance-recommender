from langchain_core.documents import Document

chunk = Document(
    page_content=('제23조 (강제집행 등으로 인하여 해지된 특별약관의 특별부활(효력회복))\n'
 '① 회사는 계약자의 해약환급금 청구권에 대한 강제집행, 담보권실행, 국세 및 지방세 체 납처분절차에 의해 특별약관이 해지된 경우 해지 '
 '당시의 보험수익자가 계약자의 동의 를 얻어 특별약관 해지로 회사가 채권자에게 지급한 금액을 회사에 지급하고 제17조 (특별약관내용의 변경 '
 '등) 제1항의 절차에 따라 계약자 명의를 보험수익자로 변경하 여 특별약관의 특별부활(효력회복)을 청약할 수 있음을 보험수익자에게 '
 '통지하여야 합 니다.\n'
 '<용어풀이>\n'
 '[강제집행과 담보권실행]'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 104},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000622',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
