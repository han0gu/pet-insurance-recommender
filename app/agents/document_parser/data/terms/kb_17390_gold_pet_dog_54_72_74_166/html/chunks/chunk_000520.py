from langchain_core.documents import Document

chunk = Document(
    page_content=('. 흡인(吸引) 차단(NERVE</td><td></td><td></td><td>사항은 '
 '제외합니다.</td></tr><tr><td></td><td></td></tr><tr><td>2. 천자(穿刺) 등의 '
 '조치</td><td></td></tr><tr><td>3. 신경(神經)</td><td>BLOCK)</td></tr><tr><td>4. '
 '미용성형 목적의</td><td>수술</td></tr><tr><td>5. 피임(避姙) 목적의 수술 6. 검사 및 진단을 위한 7'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
