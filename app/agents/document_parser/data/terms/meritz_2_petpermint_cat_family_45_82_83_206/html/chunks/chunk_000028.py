from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만,【별표2(장해분류표)】의 각 신체부<br>위별 판정기준에서 별도로 정한 경우에는 그 기준에 따릅니<br>다.</p><br><p '
 "id='36' data-category='paragraph' style='font-size:20px'>\uf000 이미 이 보장에서 "
 '후유장해보험금 지급사유에 해당되지<br>않았거나(보장개시 이전의 원인에 의하거나 또는 그 이전에<br>발생한 후유장해를 포함합니다), '
 '후유장해보험금이 지급되<br>지 않았던 피보험자에게 그 신체의 동일 부위에 또다시 제6<br>항에 규정하는 후유장해상태가 발생하였을 '
 '경우에는'),
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
