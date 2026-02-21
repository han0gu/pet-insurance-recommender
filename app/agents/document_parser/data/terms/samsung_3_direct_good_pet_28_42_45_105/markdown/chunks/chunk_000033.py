from langchain_core.documents import Document

chunk = Document(
    page_content=('회사가 계약자 또는 보험수익자 1명에 대하여 한 행위는 각각 다른 계약자 또는 보험\n'
 '수익자에게도 효력이 미칩니다.\n'
 '③ 계약자가 2명 이상인 경우에는 그 책임을 연대로 합니다.- \n'
 '# <예시안내>[계약자가 2명 이상인 경우]\n'
 '계약자가 2명 이상인 경우 계약 전 알릴 의무, 보험료 납입의무 등 보험계약에 따른 계약자의 의무\n'
 '二 ···- 31 -# <용어풀이># [연대]2인 이상이 연대하여 책임을 지므로 각자 채무의 전부를 이행할 책임을 지되(지분만큼 분할하여'),
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
