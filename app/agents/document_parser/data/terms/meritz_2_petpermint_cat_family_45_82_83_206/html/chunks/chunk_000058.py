from langchain_core.documents import Document

chunk = Document(
    page_content=("등에 연동<br>하여 일정기간마다 변동되는 이율을 말합니다.</p><h1 id='77' "
 "style='font-size:20px'>【최저보증이율】</h1><br><p id='78' data-category='paragraph' "
 "style='font-size:16px'>회사의 운용자산이익률 및 외부지표금리가 하락하더라도<br>회사에서 지급을 보증하는 최저한도의 "
 '적용이율입니다.<br>예를 들어, 계약자적립액이 [보장]공시이율에 따라 적립<br>되며 [보장]공시이율이 0.1%인 경우, '
 '계약자적립액은<br>[보장]공시이율(0.1%)이 아닌'),
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
