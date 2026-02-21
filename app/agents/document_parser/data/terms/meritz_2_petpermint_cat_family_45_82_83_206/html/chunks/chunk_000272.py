from langchain_core.documents import Document

chunk = Document(
    page_content=(". 유기동물 보호센터 등에서 사육ㆍ관리하는 고양이(猫)</td></tr></tbody></table><footer id='79' "
 "style='font-size:14px'>85</footer><h1 id='80' style='font-size:20px'>\uf000 "
 "지급사유 관련 용어</h1><br><table id='81' "
 "style='font-size:16px'><thead><tr><td>용어</td><td>정의</td></tr></thead><tbody><tr><td>상해</td><td>보험기간 "
 '중에 발생한 급격하고도 우연한 외래 의'),
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
