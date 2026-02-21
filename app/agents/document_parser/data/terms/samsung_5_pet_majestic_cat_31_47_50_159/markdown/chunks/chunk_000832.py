from langchain_core.documents import Document

chunk = Document(
    page_content=('| 7) 극심한 치매 : CDR 척도 5점 | 100 |\n'
 '| 8) 심한 치매 : CDR 척도 4점 | 80 |\n'
 '| 9) 뚜렷한 치매 : CDR 척도 3점 | 60 |\n'
 '| 10) 약간의 치매 : CDR 척도 2점 | 40 |\n'
 '| 11) 심한 뇌전증 발작이 남았을 때 | 70 |\n'
 '| 12) 뚜렷한 뇌전증 발작이 남았을 때 | 40 |\n'
 '| 13) 약간의 뇌전증 발작이 남았을 때 | 10 |\n'
 '# 나. 장해판정기준# 1) 신경계- 가) "신경계에 장해를 남긴 때" 라 함은 뇌, 척수 및 말초신경계 손상으로 "<'),
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
