from langchain_core.documents import Document

chunk = Document(
    page_content=('정당한 사유 없이 이를 거부하여서는 아니<br>된다.<br>④ 제1항부터 제3항까지의 규정에 따른 진단서, 검안서, 증명서 또는 처방전의 '
 '서<br>식, 기재사항, 그 밖에 필요한 사항은 농림축산식품부령으로 정한다.<br>⑤ 제1항에도 불구하고 농림축산식품부장관에게 신고한 '
 '축산농장에 상시고용된 수<br>의사와 「동물원 및 수족관의 관리에 관한 법률」 제8조에 따라 허가받은 동물<br>원 또는 수족관에 '
 '상시고용된 수의사는 해당 농장, 동물원 또는 수족관의 동물<br>에게 투여할 목적으로 처방대상 동물용 의약품에 대한 처방전을 발급할'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
