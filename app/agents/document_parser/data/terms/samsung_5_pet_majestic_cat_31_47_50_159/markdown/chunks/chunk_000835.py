from langchain_core.documents import Document

chunk = Document(
    page_content=('월이 지난 후에 판정함을 원칙으로 한다. 단, 질병발생 또는 상해를 입은\n'
 '후 의식상실이 1개월 이상 지속된 경우에는 질병발생 또는 상해를 입은 후\n'
 '12개월이 지난 후에 판정할 수 있다.\n'
 '나) 정신행동장해는 장해판정 직전 1년 이상 충분한 정신건강의학과의 전문적\n'
 '치료를 받은 후 치료에도 불구하고 장해가 고착되었을 때 판정하여야 하\n'
 '며, 그렇지 않은 경우에는 그로써 고정되거나 중하게 된 장해에 대해서는\n'
 '인정하지 않는다.\n'
 '다) "정신행동에 극심한 장해를 남긴 때" 라 함은 장해판정 직전 1년 이상 지'),
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
