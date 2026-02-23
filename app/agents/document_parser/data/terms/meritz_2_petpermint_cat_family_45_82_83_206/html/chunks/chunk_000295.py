from langchain_core.documents import Document

chunk = Document(
    page_content=('. 이 경우 상시고용된 수의사의 범위,<br>신고방법, 처방전 발급 및 보존 방법, 진료부 작성<br>및 보고, 교육, 준수사항 등 그 '
 "밖에 필요한 사항은<br>농림축산식품부령으로 정한다.</p><br><p id='21' data-category='paragraph' "
 "style='font-size:16px'>\uf000 제1항에도 불구하고 지정된 동물병원에서 진료를 받고<br>「동물병원 보험금 "
 '자동청구」절차를 이용한 경우에는 제1<br>항의 서류를 제출한 것으로 간주합니다'),
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
