from langchain_core.documents import Document

chunk = Document(
    page_content=('받지 않은 경우에는 최초 계약 청약일부터 5년이 지난 이후에는 이 특별약관을 적용\n'
 '하지 않습니다. 단, 계약 청약일 현재 부담보 기간을 「보험계약의 보험기간 전체」로\n'
 '적용한 유사계약이 유지중이거나, 계약 청약일 전 6개월 이내에 계약자 및 피보험자-'),
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
