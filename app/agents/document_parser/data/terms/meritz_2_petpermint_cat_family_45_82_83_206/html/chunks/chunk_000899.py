from langchain_core.documents import Document

chunk = Document(
    page_content=('발생한 날의 다음날부터 청구일까지의 기간</td><td>1년이내 : [보장]공시이율의 50%</td></tr><tr><td>1년초과기간 '
 ': [보장]공시이율의 40%</td></tr><tr><td>청구일의 다음날부터 지급일까지의 '
 "기간</td><td>보험계약대출이율</td></tr></tbody></table><br><p id='26' "
 "data-category='paragraph' style='font-size:16px'>주) 1"),
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
