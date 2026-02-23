from langchain_core.documents import Document

chunk = Document(
    page_content=('도의 계약해당일이 없는 경우에는 해당 월의 마지막 날을\n'
 '계약해당일로 합니다.| 예시1) | 계약일 : 2020년 10월 1일 -> 계약해당일 : 10월 1일 |\n'
 '| --- | --- |\n'
 '| 예시2) | 계약일 : 2020년 2월 29일 -> 계약해당일 : 2월 말일 |\n'
 '# 제25조(계약의 소멸)\uf000 제3조(보험금의 지급사유)에서 정한 일반상해80%이상후\n'
 '유장해보험금 지급사유가 발생한 경우에는 이 보장책임은\n'
 '그 때부터 소멸됩니다.\n'
 '\uf000 제1항에 따라 이 계약의 보장책임이 소멸된 때에는 회사'),
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
