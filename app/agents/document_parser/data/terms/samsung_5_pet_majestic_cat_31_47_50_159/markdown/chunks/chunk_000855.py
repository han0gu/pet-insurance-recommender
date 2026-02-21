from langchain_core.documents import Document

chunk = Document(
    page_content=('| 6. 요추 및 골반의 골절 | S32 |\n'
 '| 7. 어깨 및 위팔의 골절 | S42 |\n'
 '| 8. 아래팔의 골절 | S52 |\n'
 '| 9. 손목 및 손부위의 골절 | S62 |\n'
 '| 10. 대퇴골의 골절 | S72 |\n'
 '| 11. 발목을 포함한 아래다리의 골절 | S82 |\n'
 '| 12. 발목을 제외한 발의 골절 | S92 |\n'
 '| 13. 여러 신체부위를 침범한 골절 | T02 |\n'
 '| 14. 척추의 상세불명 부위의 골절 | T08 |\n'
 '| 15. 팔의 상세불명 부위의 골절 | T10 |\n'
 '| 16. 다리의 상세불명 부위의 골절 | T12 |'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
