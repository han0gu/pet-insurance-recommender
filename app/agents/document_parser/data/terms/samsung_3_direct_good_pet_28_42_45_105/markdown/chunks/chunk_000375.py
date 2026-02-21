from langchain_core.documents import Document

chunk = Document(
    page_content=('- 인이 보험수익자로 지정된 경우에는 제1항의 통지를 계약자에게 할 수 있습니다.\n'
 '<용어풀이># [법정상속인]피상속인의 사망에 의하여 민법의 규정에 의한 상속순위에 따라 상속받는 자를 말합니다.※ 상속순위- ① '
 '피상속인의 직계비속\n'
 '- ③ 피상속인의 형제자매\n'
 '- ② 피상속인의 직계존속\n'
 '- ④ 피상속인의 4촌 이내의 방계혈족\n'
 '[직계비속]\n'
 '자기로부터 직계로 이어져 내려가는 혈족. 자녀, 손자, 증손 등\n'
 '[직계존속]\n'
 '조상으로부터 직계로 내려와 자기에 이르는 사이의 혈족. 부모, 조부모 등\n'
 '[방계혈족]'),
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
