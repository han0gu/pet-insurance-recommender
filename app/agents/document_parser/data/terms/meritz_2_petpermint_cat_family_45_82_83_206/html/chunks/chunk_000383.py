from langchain_core.documents import Document

chunk = Document(
    page_content=('날의 다음<br>날에 계약이 해지된다는 내용(이 경우 계약이 해지되<br>는 때에는 즉시 해약환급금에서 보험계약대출원금과<br>이자가 '
 "차감된다는 내용을 포함합니다)</p><br><h1 id='32' style='font-size:20px'>【 납입최고(독촉) "
 "】</h1><br><p id='33' data-category='paragraph' style='font-size:16px'>약정된 "
 "기일까지 보험료가 납입되지 않을 경우, 회사가<br>계약자에게 납입을 재촉하는 것을 말합니다.</p><br><p id='34'"),
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
