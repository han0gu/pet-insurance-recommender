from langchain_core.documents import Document

chunk = Document(
    page_content=('| --- | --- | --- |\n'
 '| 는 계약을 말합니다. 우체국, 신협, 새마을금고 등이 공제계약을 취급합니다. | 는 계약을 말합니다. 우체국, 신협, 새마을금고 등이 '
 '공제계약을 취급합니다. | 는 계약을 말합니다. 우체국, 신협, 새마을금고 등이 공제계약을 취급합니다. |\n'
 '# 제12조(손해방지의무)계약자 또는 피보험자는 아래의 사항을 이행하여야 합\uf000 보험사고가 생긴 때에는니다.\n'
 '1. 손해의 방지 또는 경감을 위하여 노력하는 일(피해자에 대한 응급처치, 긴급- 122 -- 호송 또는 그 밖의 긴급조치를 포함합니다)'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
