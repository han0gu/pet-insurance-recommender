from langchain_core.documents import Document

chunk = Document(
    page_content=("신경증 및 각종 인격장애는 보상의 대상이<br>되지 않는다.</p><h1 id='46' style='font-size:20px'>3) "
 "치매</h1><br><p id='47' data-category='paragraph' style='font-size:20px'>가) "
 '“치매”라 함은 정상적으로 성숙한 뇌가 질병이<br>나 외상 후 기질성 손상으로 파괴되어 한번 획득<br>한 지적기능이 지속적 또는 '
 "전반적으로 저하되는<br>것을 말한다.</p><br><p id='48' data-category='paragraph'"),
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
