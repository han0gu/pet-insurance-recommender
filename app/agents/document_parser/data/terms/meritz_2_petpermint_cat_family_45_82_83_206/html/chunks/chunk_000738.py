from langchain_core.documents import Document

chunk = Document(
    page_content=('하에 전산화단층촬<br>영(CT)을 사용하는 촬영 의료행위를 말합니다.<br>\uf000 제1항의 내시경처치라 함은 제1조(보험금의 '
 '지급사유)에<br>서 정한 수의사에 의하여 진단 및 치료가 필요하다고 인정<br>된 경우로서 수의사의 관리 하에 내시경을 이용하여 '
 "비침습<br>적으로 시행하는 의료행위를 말하며, 식도, 위 또는 장에<br>시행하는 경우에 한합니다.</p><br><h1 id='47' "
 "style='font-size:16px'>【자기공명영상(MRI)】</h1><br><p id='48' "
 "data-category='paragraph'"),
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
