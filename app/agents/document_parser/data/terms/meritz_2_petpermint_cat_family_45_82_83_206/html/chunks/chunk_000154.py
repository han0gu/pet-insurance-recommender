from langchain_core.documents import Document

chunk = Document(
    page_content=("시행령 제44조의2(타인의 생명보험)】</p><footer id='7' style='font-size:14px'>67</footer><p "
 "id='8' data-category='paragraph' style='font-size:20px'>법 제731조제1항에 따른 본인 확인 "
 '및 위조ㆍ변조 방지에<br>대한 신뢰성을 갖춘 전자문서는 다음 각 호의 요건을 모<br>두 갖춘 전자문서로 한다.</p><br><p '
 "id='9' data-category='list' style='font-size:16px'>1"),
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
