from langchain_core.documents import Document

chunk = Document(
    page_content=("- 공휴일에 관한 규정'에 따른 공휴일과 노동절을 제외합니다.\n"
 '# ⑤ 보험료 관련 용어1. 보험료 : 손해를 보장하는데 필요한 보험료를 말합니다.⑥ [갱신형] 특별약관의 갱신 관련 용어- 1. '
 '최초계약: [갱신형] 특별약관이 최초로 부가되는 경우를 말합니다.\n'
 '- 2. 갱신계약 : [갱신형] 특별약관의 보험기간이 끝난 후 제도성 특별약관 「4-1.\n'
 '- [갱신형] 특별약관의 자동갱신 특별약관」 에 따라 갱신된 경우를 말합니다.\n'
 '- 3. 갱신일: [갱신형] 특별약관이 갱신되기 직전 계약(이하 「갱신 전 계약」 이라 합니'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
