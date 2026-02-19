from langchain_core.documents import Document

chunk = Document(
    page_content=('제 33조의2 (위법계약의 해지)\n'
 '① 계약자는 「금융소비자 보호에 관한 법률」제47조 및 관련규정이 정하는 바에 따라 계약체결에 대한 회사의 법위반사항이 있는 경우 '
 '계약체결일부터 5년 이내의 범위에 서 계약자가 위반사항을 안 날부터 1년 이내에 계약해지요구서에 증빙서류를 첨부하 여 위법계약의 해지를 '
 '요구할 수 있습니다.\n'
 '<용어풀이>\n'
 '[위법계약]\n'
 "금융상품판매업자등이 '금융소비자보호에 관한 법률' 제47조에서 정한 적합성원칙, 적정성원칙, 설 명의무, 불공정영업행위의 금지 또는 "
 '부당권유행위 금지를 위반한 계약을 말합니다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 44},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000148',
              'chunk_char_len': 292,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
