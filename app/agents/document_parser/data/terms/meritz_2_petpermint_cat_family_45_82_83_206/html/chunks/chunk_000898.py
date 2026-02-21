from langchain_core.documents import Document

chunk = Document(
    page_content=('61일 이후부터 90일 이내 기간</td><td>보험계약대출이율 + 가산이율(6.0%)</td></tr><tr><td>지급기일의 91일 '
 '이후 기간</td><td>보험계약대출이율 + 가산이율(8.0%)</td></tr><tr><td rowspan="3">만기환급금(보통약관 '
 '제10조 제1항) 및 해약환급금 (보통약관 제35조 제1항) (특별약관이 부가된 경우 특별약관의 해약환급금 포함)</td><td '
 'rowspan="2">지급사유가 발생한 날의 다음날부터 청구일까지의 기간</td><td>1년이내 : [보장]공시이율의'),
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
