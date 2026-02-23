from langchain_core.documents import Document

chunk = Document(
    page_content=('남긴 때”라 함은 아<br>래의 경우 중 하나 이상에 해당되는 때를 말한다.<br>가) 언어평가상 자음정확도가 75%미만인 '
 '경우<br>나) 언어평가상 표현언어지수 65 미만인 경우<br>10) 말하는 기능의 장해는 1년 이상 지속적인 언어치료를<br>시행한 후 '
 '증상이 고착되었을 때 평가하며, 객관적인<br>검사를 기초로 평가한다.<br>11) 뇌‧중추신경계 손상(정신‧인지기능 저하, 편마비 '
 '등)<br>으로 인한 말하는 기능의 장해(실어증, 구음장애)<br>또는 씹어먹는 기능의 장해는 신경계‧정신행동 장해<br>평가와 비교하여 '
 '그'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
