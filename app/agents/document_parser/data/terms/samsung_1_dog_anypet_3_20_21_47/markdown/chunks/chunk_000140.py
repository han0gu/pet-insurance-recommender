from langchain_core.documents import Document

chunk = Document(
    page_content=('험 계약상의 모든 권리, 의무를 행사할 수 있어야 합니다.# 1. 제1종 단체동일한 회사, 사업장, 관공서, 국영기업체, 조합 등 5인 '
 '이상의 근로자를 고용하고 있는 단체.\n'
 '다만, 사업장, 직제, 직종 등으로 구분되어 있는 경우의 단체소속 여부는 관련법규 등에서 정하\n'
 '는 바에 따릅니다.- 2. 제2종 단체\n'
 '- 비영리법인단체 또는 변호사회, 의사회등 동업자단체로서 5인 이상의 구성원이 있는 단체\n'
 '- 3. 제3종 단체\n'
 '- 그밖에 단체의 구성원을 확정시킬 수 있고 계약의 일괄적인 관리가 가능한 단체로서 5인 이상'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
