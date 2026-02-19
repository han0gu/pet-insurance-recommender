from langchain_core.documents import Document

chunk = Document(
    page_content=('단체계약 특별약관\n'
 '제1조(계약의 적용 범위)\n'
 '① 피보험자가 다음 중 한가지의 단체에 소속되어야 하며, 단체를 대표하여 계약자로 된 자가 단체보 험 계약상의 모든 권리, 의무를 행사할 '
 '수 있어야 합니다.\n'
 '1. 제1종 단체\n'
 '동일한 회사, 사업장, 관공서, 국영기업체, 조합 등 5인 이상의 근로자를 고용하고 있는 단체. 다만, 사업장, 직제, 직종 등으로 '
 '구분되어 있는 경우의 단체소속 여부는 관련법규 등에서 정하 는 바에 따릅니다.'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 35},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000172',
              'chunk_char_len': 240,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
