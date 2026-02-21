from langchain_core.documents import Document

chunk = Document(
    page_content=('- ② 갱신일에 있어서 반려동물의 만나이가 회사가 정한 나\n'
 '- 이의 범위내일 것\n'
 '- ③ 갱신전 계약의 보험료가 정상적으로 납입완료 되었을\n'
 '- 것\n'
 '- ④ 갱신전 계약이 소멸되지 않을 것\n'
 '\uf000 갱신계약의 보험기간은 갱신전 계약의 보험기간과 동일\n'
 '한 것으로 합니다. 다만, 갱신일의 반려동물의 만나이로부\n'
 '터 갱신종료만나이(갱신시점의 갱신종료만나이를 말합니다)\n'
 '까지의 기간이 갱신전 계약의 보험기간 미만인 경우에는 그\n'
 '잔여기간을 보험기간으로 합니다.\n'
 '\uf000 회사는 갱신계약에 대하여 갱신전 약관을 적용하며, 보'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
