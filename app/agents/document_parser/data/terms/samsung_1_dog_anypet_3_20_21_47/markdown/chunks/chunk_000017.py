from langchain_core.documents import Document

chunk = Document(
    page_content=('- 소요법, 면역요법 등의 대체적 처치에 의한 치료를 위한 비용\n'
 '- 6 -당신에게 좋은보험 삼성화재- 11. 가입동물의 이송비, 마이크로칩의 삽입 비용, 안락사를 위한 비용, 장례식비용, 매장비용 등 '
 '가\n'
 '- 입동물의 사망 후에 소요된 비용, 각종 증명서류의 작성비용(운송비 포함), 카운슬링 비용\n'
 '- 12. 산후 문제행동, 수유에 따르는 칼슘 부족에 의한 경련 및 기타 임신 · 출산과 관련된 질병 치료\n'
 '- 에 대한 비용\n'
 '- 13. 슬개골탈구, 십자인대파열, 고관절탈구(고관절형성부전, 대퇴골두 괴사증으로 인한 탈구 포함)와'),
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
