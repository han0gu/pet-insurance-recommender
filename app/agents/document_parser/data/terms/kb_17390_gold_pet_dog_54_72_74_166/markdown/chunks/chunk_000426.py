from langchain_core.documents import Document

chunk = Document(
    page_content=('\uf000 피보험자가 보험기간 중 사망하고, 그 후에 제3조(특정정신질환의 정의 및 진단\n'
 '확정)에서 정한 "특정정신질환"을 직접적인 원인으로 사망한 사실이 확인된 경우\n'
 '에는 그 사망일을 진단 확정일로 보고 제1조(보험금의 지급사유)에 해당하는 경\n'
 '우에 한하여 해당 보험금을 지급합니다. 다만, 제4조(특별약관의 소멸)에 따라- \n'
 '# 제5조(준용규정)94 KB 금쪽같은 펫보험(강아지)(무배당)(26.01)이 특별약관에서 정하지 않은 사항은 보통약관 제1절 일반조항을 '
 '따릅니다. 다만,'),
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
