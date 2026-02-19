from langchain_core.documents import Document

chunk = Document(
    page_content=('8 | 췌장\n'
 '9 | 비장\n'
 '10 | 기관, 기관지, 폐, 흉막 및 흉곽(늑골 포함)\n'
 '11 | 코[외비(코 바깥), 비강(코 안) 및 부비강(코 곁굴) 포함]\n'
 '12 | 인두 및 후두(편도 포함)\n'
 '13 | 식도\n'
 '14 | 구강, 치아, 혀, 악하선(턱밑샘), 이하선(귀밑샘) 및 설하선(혀밑샘)\n'
 '15 | 귀[외이(바깥 귀), 고막, 중이(가운데귀), 내이(속귀), 청신경 및 유양돌기(꼭지돌기)포함]\n'
 '16 | 안구 및 안구부속기[안검(눈꺼풀), 결막, 누기(눈물샘), 안근 및 안와내조직 포함]\n'
 '17 | 신장\n'
 '18 | 부신'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 102},
 'term_type': 'special',
 'clause': {'clause_type': 'other',
            'risk_domains': ['digestive', 'dental', 'urinary', 'eye', 'other']},
 'indexing': {'chunk_id': 'chunk_000662',
              'chunk_char_len': 288,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
