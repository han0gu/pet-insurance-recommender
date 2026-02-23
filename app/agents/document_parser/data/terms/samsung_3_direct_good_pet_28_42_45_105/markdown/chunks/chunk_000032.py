from langchain_core.documents import Document

chunk = Document(
    page_content=('조상으로부터 직계로 내려와 자기에 이르는 사이의 혈족. 부모, 조부모 등\n'
 '[방계혈족]\n'
 '자기의 형제자매와 형제자매의 직계비속, 직계존속의 형제자매 및 그 형제자매의 직계비속# 제12조 (대표자의 지정)① 계약자 또는 '
 '보험수익자가 2명 이상인 경우에는 각 대표자를 1명 지정하여야 합니다.\n'
 '이 경우 그 대표자는 각각 다른 계약자 또는 보험수익자를 대리하는 것으로 합니다.\n'
 '② 지정된 계약자 또는 보험수익자의 소재가 확실하지 않은 경우에는 이 계약에 관하여\n'
 '회사가 계약자 또는 보험수익자 1명에 대하여 한 행위는 각각 다른 계약자 또는 보험'),
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
