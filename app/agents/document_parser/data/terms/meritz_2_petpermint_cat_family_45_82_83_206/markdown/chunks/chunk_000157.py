from langchain_core.documents import Document

chunk = Document(
    page_content=('- 폐사 진단서는 다른 수의사에게서 발급받을 수 있다.\n'
 '- ③ 수의사는 직접 진료하거나 검안한 동물에 대한 진단\n'
 '- 서, 검안서, 증명서 또는 처방전의 발급을 요구받았\n'
 '- 을 때에는 정당한 사유 없이 이를 거부하여서는 아니\n'
 '- 된다.\n'
 '- ④ 제1항부터 제3항까지의 규정에 따른 진단서, 검안서,\n'
 '- 증명서 또는 처방전의 서식, 기재사항, 그 밖에 필요\n'
 '- 한 사항은 농림축산식품부령으로 정한다.\n'
 '- ⑤ 제1항에도 불구하고 농림축산식품부장관에게 신고한\n'
 '- 축산농장에 상시고용된 수의사는 해당 농장의 가축에'),
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
