from langchain_core.documents import Document

chunk = Document(
    page_content=('- 등기우편 등 우편물에 대한 기록이 남는 방법으로 회사가 알린 사항은 일반적으로 도달에 필요한\n'
 '- 기간이 지난 때에 계약자 또는 피보험자에게 도달된 것으로 봅니다.\n'
 '# 제14조(사기에 의한 계약)계약자, 피보험자 또는 이들의 대리인의 사기에 의하여 계약이 성립되었음을 회사가 증명하는 경우에\n'
 '는 계약일부터 5년 이내(사기사실을 안 날부터 1개월 이내)에 계약을 취소할 수 있습니다.제4관 보험계약의 성립과 유지제15조(보험계약의 '
 '성립)- ① 계약은 계약자의 청약과 회사의 승낙으로 이루어집니다.'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
