from langchain_core.documents import Document

chunk = Document(
    page_content=('야 합니다.# 제3조(준용규정)이 추가특별약관에 정하지 않은 사항은 보통약관 및 보험료자동납입 특별약관을 따릅니다.- 36 -# 단체계약 '
 '특별약관# 제1조(계약의 적용 범위)① 피보험자가 다음 중 한가지의 단체에 소속되어야 하며, 단체를 대표하여 계약자로 된 자\n'
 '가 단체보험 계약상의 모든 권리, 의무를 행사할 수 있어야 합니다.# 1. 제1종 단체동일한 회사, 사업장, 관공서, 국영기업체, 조합 '
 '등 5인 이상의 근로자를 고용하고\n'
 '있는 단체. 다만, 사업장, 직제, 직종 등으로 구분되어 있는 경우의 단체소속 여부는'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000175',
              'chunk_char_len': 289,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
