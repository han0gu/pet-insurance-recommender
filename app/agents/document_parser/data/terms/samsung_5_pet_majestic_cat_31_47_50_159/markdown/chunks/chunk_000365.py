from langchain_core.documents import Document

chunk = Document(
    page_content=('합니다) 중에 상해를 입고 병원 또는 의원(한방병원 또는 한의원을 포함합니다) 등에\n'
 '서 치료를 받고 그 직접적인 결과로써 안면부에 외형상의 반흔(흉터)이나 추상(추한\n'
 '모습)장해, 신체의 기형이나 기능장해가 발생하여 그 원상회복(이하 「안면부 상해흉\n'
 '터복원」이라 합니다)을 목적으로 사고일부터 2년 이내에 성형외과 전문의로부터 안\n'
 '면부에 5cm이상의 성형수술(단, 사고발생시점 만 15세 미만자의 경우 부득이 사고일\n'
 '부터 2년이 지난 후에 성형수술이 가능하다는 진단을 받은 경우에는 그 진단으로 대'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
