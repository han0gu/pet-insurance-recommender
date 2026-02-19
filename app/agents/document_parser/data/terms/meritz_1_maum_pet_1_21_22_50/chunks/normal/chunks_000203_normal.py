from langchain_core.documents import Document

chunk = Document(
    page_content=('단체계약 특별약관\n'
 '제1조(계약의 적용 범위)\n'
 '① 피보험자가 다음 중 한가지의 단체에 소속되어야 하며, 단체를 대표하여 계약자로 된 자 가 단체보험 계약상의 모든 권리, 의무를 행사할 '
 '수 있어야 합니다.\n'
 '1. 제1종 단체\n'
 '동일한 회사, 사업장, 관공서, 국영기업체, 조합 등 5인 이상의 근로자를 고용하고 있는 단체. 다만, 사업장, 직제, 직종 등으로 '
 '구분되어 있는 경우의 단체소속 여부는 관련법규 등에서 정하는 바에 따릅니다.\n'
 '2. 제2종 단체\n'
 '비영리법인단체 또는 변호사회, 의사회 등 동업자단체로서 5인 이상의 구성원이 있는 단체'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 37},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000203',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
