from langchain_core.documents import Document

chunk = Document(
    page_content=('| 51 | 담석증 | K80 | 담석증 |\n'
 '| 52 | 요로결석증 | N20 | 신장 및 요관의 결석 |\n'
 '| 52 | 요로결석증 | N21 | 하부요로의 결석 |\n'
 '| 52 | 요로결석증 | N22 | 달리 분류된 질환에서의 요로의 결석 |\n'
 '| 52 | 요로결석증 | N23 | 상세불명의 신장 급통증 |\n'
 '| 53 | 관절증 및 류마티스 관절염 | M05 | 혈청검사양성 류마티스관절염 |\n'
 '| 53 | 관절증 및 류마티스 관절염 | M06 | 기타 류마티스관절염 |'),
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
