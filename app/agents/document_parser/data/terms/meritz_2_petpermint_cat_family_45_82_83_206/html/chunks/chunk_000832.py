from langchain_core.documents import Document

chunk = Document(
    page_content=('유의성<br>있게 확인된 경우 등과 같이 회사가 정한 기준에 따라 직접<br>관련이 있는 특정질병으로 제한하며, 부담보 설정 범위 '
 '및<br>사유를 계약자에게 설명하여 드립니다.<br>\uf000 이 특별약관의 보장개시일은 보통약관 제26조(제1회 보<br>험료 및 '
 '회사의 보장개시)에서 정한 보장개시일과 동일합니<br>다.<br>\uf000 계약이 해지, 기타사유에 따라 효력이 없는 경우에는 '
 '이<br>특별약관도 더 이상 효력이 없습니다.<br>\uf000 이 특별약관에서 정한 보장개시일 이전에 발생한 질병에<br>대하여 계약을 '
 '무효로 하는 경우에도'),
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
